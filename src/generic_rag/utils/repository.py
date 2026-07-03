from generic_rag.db.session import get_current_session


class RepositoryMixin[EntityT]:
    @staticmethod
    async def delete(entity: EntityT):
        await get_current_session().delete(entity)

    @staticmethod
    async def save(entity: EntityT) -> EntityT:
        session = get_current_session()

        is_new_object = entity not in session.dirty and entity not in session.new

        if is_new_object:
            session.add(entity)

        if session.is_modified(entity):
            await session.flush()
            await session.refresh(entity)

        return entity
